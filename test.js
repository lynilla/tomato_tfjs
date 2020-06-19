var classNames = [];
var model;

/*
load the class names 
*/
async function loadDict() {
  
    loc = 'model/class_name.txt'
    await $.ajax({
        url: loc,
        dataType: 'text',
    }).done(success);
}

/*
load the class names
*/
function success(data) {
    const lst = data.split(/\n/)
    for (var i = 0; i < lst.length - 1; i++) {
        let symbol = lst[i]
        classNames[i] = symbol
        // console.log(classNames)	    
    }
}
/*
get the the class names 
*/
function getClassNames(indices) {
    var outp = []
    for (var i = 0; i < indices.length; i++)
        outp[i] = classNames[indices[i]]
    // console.log(outp)	
    return outp
}
/*
find predictions
*/
function findTopValues(inp, count) {
    var outp = [];
    let indices = findIndicesOfMax(inp, count)
    // show  scores
    for (var i = 0; i < indices.length; i++)
        outp[i] = inp[indices[i]]
    return outp
}
/*
get indices of the top probs
*/
function findIndicesOfMax(inp, count) {
    var outp = [];
    for (var i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) {
                return inp[b] - inp[a];
            }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}
function preprocess(img)
{
    console.log("================Preprocessing Start=====================");
    //convert the image data to a tensor 
    let tensor = tf.browser.fromPixels(img)
    //resize to 50 X 50
    const resized = tf.image.resizeBilinear(tensor, [256, 256]).expandDims(0).toFloat()
    // Normalize the image 
    const offset = tf.scalar(255.0);
    const normalized = resized.div(offset);
    //We add a dimension to get a batch shape 
    // batched = resized / 255
    // const batched2 = normalized.expandDims(0)


    console.log(normalized.print())
    console.log(normalized.shape)
    console.log("================Preprocessing End=====================");
    return normalized

}
/*
get the prediction 
*/
function predict(imgData) {
        
        console.log("================Predicting Start=====================");
        //get the prediction 
        var pred = model.predict(preprocess(imgData)).dataSync()
        $(".loader").hide();
        console.log("================Predicting End=====================");    

        console.log(pred)            
        //retreive the highest probability class label 
        const idx = tf.argMax(pred);

                
        //find the predictions 
        var indices = findIndicesOfMax(pred, 1)
        // console.log(indices)
        var probs = findTopValues(pred, 1)
        var names = getClassNames(indices) 

        //set the table 
        //setTable(names, probs) 
        document.getElementById("result-name").innerHTML = " - " + names;
        document.getElementById("result-prob").innerHTML = " - " + probs;
        console.log(names);
        
  }

async function start(){
    $("#result-name").html("");
    $("#result-prob").html("");
    $(".loader").show();
    //img = document.getElementById('image').files[0];
    var status = document.getElementById('status')      

    status.innerHTML = 'Loading Model .....'
    
    tf.ENV.set("WEBGL_PACK", false);

    model = await tf.loadLayersModel('http://localhost:8080/model/model.json')
    
    // console.log(model.getWeights()[0].print())
    status.innerHTML = 'Model Loaded'     

    img = document.getElementById('list').firstElementChild.firstElementChild;
    console.log(img)
        
	//load the class names
    await loadDict()
    predict(img)
         
}
